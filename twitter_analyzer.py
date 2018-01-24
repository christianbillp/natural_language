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
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


class TwitterAnalyzer():
    '''Main class for TwitterAnalyzer. Construct with database filename'''

    sw = stopwords.words('english')
    sw = sw + [char for char in string.punctuation]
    sw = sw + ['\'s', 'RT', '\'\'', 'The', '``', '’', '“', "n't", 'http', 'https', 'I',' you', 'amp']

    def __init__(self):       
        pass
        
    def get_mongo(self, collection='twitter'):
        df = pd.DataFrame([])

        with MongoClient('mongodb://twitter:aabbccdd@deuscortex.com:27017/nlp') as client:
            data = [entry for entry in client['nlp'][collection].find()]

        # Fixed names without need for preprocessing            
        for value in ['lang', 'text', 'created_at', 'id']:
            df[value] = [entry[value] for entry in data]

        # Documents needing preprossing
        df['user_id'] = [entry['user']['id'] for entry in data]
        df['screen_name'] = [entry['user']['screen_name'] for entry in data]
        df['hashtags'] = [[hashtag['text'].lower() for hashtag in item] for item in [entry['entities']['hashtags'] for entry in data]];
        self.df = df

        # Document needing calculations
        df['polarity'] = [TextBlob(tweet).sentiment[0] for tweet in self.df['text']]
        df['subjectivity'] = [TextBlob(tweet).sentiment[1] for tweet in self.df['text']]
        df['tokens'] = df['text'].apply(self.clean_tokenize)
        df['tokens'] = df['tokens'].apply(lambda x: [item for item in x if item not in ta.sw])
        df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word.startswith('/') == False])
        df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word.startswith("'") == False])
        
        
        df = df.drop_duplicates('text')
        
        df = df[df['lang'] == 'en']
        self.df = df

        return df
    
    def clean_tokenize(self, input_tweet):
        '''Removes tags and hyperlinks from tweet'''
        wl = [word for word in input_tweet.split(' ') if word not in self.sw]
        rl = nltk.word_tokenize(' '.join(wl))

        return rl
    
    def show_sentiment(self, hashtag):
        '''Generates graph for sentiment analysis'''
        self.df['search'] = self.df['hashtags'].apply(lambda x: True if hashtag in x else False)
        x = self.df[self.df['search'] == True]['polarity'].astype(float)
        y = self.df[self.df['search'] == True]['subjectivity'].astype(float)
        center = self.find_kmeans()[0]
        plt.xlim(-1, 1)
        plt.ylim(0, 1)
        plt.title(f'Sentiment Analysis for tag: {hashtag}\nKMeans center: {self.find_kmeans()[0]}')
        plt.xlabel('Polarity')
        plt.ylabel('Subjectivity')
        plt.grid()
        plt.scatter(x, y, marker='.')
        plt.scatter(center[0], center[1], marker=(5, 2), s=300)
        
        

    def find_kmeans(self):
        '''Finds cluster center - used for sentiment analysis'''
        x = self.df[self.df['search'] == True]['polarity'].astype(float)
        y = self.df[self.df['search'] == True]['subjectivity'].astype(float)
        values = np.array(list(zip(x, y)))
        kmeans = KMeans(n_clusters=1, random_state=0).fit(values)

        return kmeans.cluster_centers_

    def compare_tags(self, tags):
        '''Compare sentiment of two tags'''
        labels = 'Pos', 'Neg'
        n_tags = len(tags)
        data = [0, 0]
        sizes = [0, 0]
        fig1, ax = plt.subplots(1, n_tags)

        for i in range(n_tags):
            self.df['search'] = self.df['hashtags'].apply(lambda x: True if tags[i] in x else False)
            data[i] = self.df[self.df['search'] == True]['polarity'].astype(float)
            positive = len(data[i][data[i] > 0])
            negative = len(data[i][data[i] < 0])
            sizes[i] = [positive, negative]
            ax[i].pie(sizes[i], labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
            ax[i].axis('equal')
            ax[i].set_title('{} for: {}'.format('polarity'.capitalize(), tags[i]))
        plt.show()

    def show_concordance(self, n, hashtag, term):
        '''Displays the context a [term] appears in'''
        self.df['search'] = self.df['hashtags'].apply(lambda x: True if hashtag in x else False)
        ts = self.df[self.df['search'] == True]['text'].values

        for i in range(n):
            text = nltk.Text(nltk.word_tokenize(ts[i]))
            print(text.concordance(term))
    


    def show_frequent_words(self, n, hashtag):
        '''Shows [n] most frequent words in corpus with [tag]'''
        # Additional stopwords may be needed depending on topic


        
        self.df['search'] = self.df['hashtags'].apply(lambda x: True if hashtag in x else False)
        ts = self.df[self.df['search'] == True]['tokens'].values
        rl = [word for wordlist in ts for word in wordlist if word not in self.sw]
        text = nltk.word_tokenize(' '.join(rl))
        fdist = nltk.FreqDist(text)
        plt.title('{} Most frequent words in tweets with hashtag: "{}"'.format(n, hashtag))
        fdist.plot(n)

        return fdist.most_common(n)
    
    def hashtag_association_mining(self, min_support=0.05):
        '''Does association mining for hashtags'''
        w = pd.DataFrame(self.df['hashtags'])
        w['hashtags'] = w['hashtags'].apply(' '.join)

        vectorizer = TfidfVectorizer(stop_words='english',
                             binary=False,
                             max_df=0.95, 
                             min_df=0.011,
                             ngram_range=(1,2),
                             use_idf=False,
                             norm=None)
        
        doc_vectors = vectorizer.fit_transform(w['hashtags'])    
        encoded = pd.DataFrame(doc_vectors.toarray(), columns=vectorizer.get_feature_names())
        frequent_itemsets = apriori(encoded, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=1)
    
        return rules

    def text_association_mining(self, min_support=0.05):
        '''Does association mining for hashtags'''
        w = pd.DataFrame(self.df['tokens'])
        w['tokens'] = w['tokens'].apply(' '.join)

        vectorizer = TfidfVectorizer(stop_words='english',
                             binary=False,
                             max_df=0.95, 
                             min_df=0.011,
                             ngram_range=(1,2),
                             use_idf=False,
                             norm=None)
        
        doc_vectors = vectorizer.fit_transform(w['tokens'])  
        
        encoded = pd.DataFrame(doc_vectors.toarray(), columns=vectorizer.get_feature_names())
        frequent_itemsets = apriori(encoded, min_support=min_support, use_colnames=True)
#        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    
        return frequent_itemsets

    def find_itemset(self, itemset):
        pass

if __name__ == '__main__':
    '''Tests generation of scatter/center plot for tag and pie chart compare'''

    ta = TwitterAnalyzer()
    df = ta.get_mongo('twitter')
    # %%
    ta.show_sentiment('trump')
    # %%
    ta.show_sentiment('hillary')
    # %%
    ta.compare_tags(['trump', 'hillary'])
    # %%
    ta.show_concordance(10, 'trump', 'trump')
    # %%
    ta.show_frequent_words(10, 'spacex')
    # %%
    hh=ta.hashtag_association_mining(0.03)
    # %%
    ll=ta.text_association_mining(0.03)
# %% End of file
