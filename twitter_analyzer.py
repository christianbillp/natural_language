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

class TwitterAnalyzer():
    '''Main class for TwitterAnalyzer. Construct with database filename'''

    def __init__(self, filename):
        self.df = pd.read_pickle(filename)

    def show_sentiment(self, tag):
        '''Generates graph for sentiment analysis'''
        x = self.df[self.df['tag'] == tag]['polarity'].astype(float)
        y = self.df[self.df['tag'] == tag]['subjectivity'].astype(float)
        center = self.find_kmeans(tag)[0]
        plt.xlim(-1, 1)
        plt.ylim(0, 1)
        plt.title('Sentiment Analysis for tag: {}\nKMeans center: {}'.format(tag, self.find_kmeans(tag)[0]))
        plt.xlabel('Polarity')
        plt.ylabel('Subjectivity')
        plt.grid()
        plt.scatter(x, y, marker='.')
        plt.scatter(center[0], center[1], marker=(5, 2), s=300)

    def find_kmeans(self, tag):
        '''Finds cluster center - used for sentiment analysis'''
        x = self.df[self.df['tag'] == tag]['polarity'].astype(float)
        y = self.df[self.df['tag'] == tag]['subjectivity'].astype(float)
        values = np.array(list(zip(x, y)))
        kmeans = KMeans(n_clusters=1, random_state=0).fit(values)

        return kmeans.cluster_centers_

    def compare_tags(self, tags, aspect):
        '''Compare sentiment of two tags'''
        labels = 'Pos', 'Neg'
        n_tags = len(tags)
        data = [0, 0]
        sizes = [0, 0]
        fig1, ax = plt.subplots(1, n_tags)

        for i in range(n_tags):
            data[i] = self.df[self.df['tag'] == tags[i]][aspect].astype(float)
            positive = len(data[i][data[i] > 0])
            negative = len(data[i][data[i] < 0])
            sizes[i] = [positive, negative]
            ax[i].pie(sizes[i], labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
            ax[i].axis('equal')
            ax[i].set_title('{} for: {}'.format(aspect.capitalize(), tags[i]))
        plt.show()

    def show_concordance(self, n, tag, term):
        '''Displays the context a [term] appears in'''
        ts = self.df[self.df['tag'] == tag]['tweet'].values

        for i in range(n):
            text = nltk.Text(nltk.word_tokenize(ts[i]))
            print(text.concordance(term))

    def show_frequent_words(self, n, tag):
        '''Shows [n] most frequent words in corpus with [tag]'''
        # Additional stopwords may be needed depending on topic
        sw = stopwords.words('english')
        sw = sw + [char for char in string.punctuation]
        sw = sw+ ['\'s', 'RT', '\'\'', 'The', '``', '’', '“', "n't", 'http', 'https', 'I',
                  tag.capitalize(), tag.lower(), tag]

        ts = self.df[self.df['tag'] == tag]['clean tokens'].values
        rl = [word for wordlist in ts for word in wordlist if word not in sw]
        text = nltk.word_tokenize(' '.join(rl))
        fdist = nltk.FreqDist(text)
        plt.title('{} Most frequent words in tweets with tag: "{}"'.format(n, tag))
        fdist.plot(n)

        return fdist.most_common(n)

    def process_sentiment(self, filename):
        '''Sentiment might not work for Danish'''
        tdf = pd.read_pickle(filename)
        tdf['polarity'] = [TextBlob(content).sentiment[0] for content in tdf['Content']]
        tdf['subjectivity'] = [TextBlob(content).sentiment[1] for content in tdf['Content']]
        tdf.to_pickle(filename)

        return tdf

if __name__ == '__main__':
    '''Tests generation of scatter/center plot for tag and pie chart compare'''

    ta = TwitterAnalyzer('twitter_db.pickle')
    temp_df = ta.df
    # %%
    ta.show_sentiment('trump')
    # %%
    ta.show_sentiment('hillary')
    # %%
    ta.compare_tags(['trump', 'hillary'], aspect='polarity')
    # %%
    ta.show_concordance(10, 'trump', 'Trump')
    # %%
    ta.show_frequent_words(10, 'trump')

    # %%

# %% End of file
