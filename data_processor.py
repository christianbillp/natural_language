# DataProcessor
# Purpose: Maintains data from data sources and main database
# Details: Data engineering and duplicate removal
#
# %% Imports and definitions
import pandas as pd
from os import listdir
pd.set_option('max_colwidth', 400)
from textblob import TextBlob
from sqlalchemy import create_engine
import pymysql

username = 'sensor'
password = 'aabbccdd'
hostname = '0ohm.dk'
db_name = 'nlp'

class DataProcessor():
    '''Universal data processor for language data'''

    def __init__(self):
        pass

    def process_twitter(self, source):
        self.df = pd.read_pickle(source)
        target_temp = pd.read_pickle('twitter_db.pickle')
        
        self.df['hashtags'] = ([[tag['text'] for tag in ent['hashtags']] for ent in self.df['entities']])
#        self.df = self.df.drop(['entities'], axis=1)
        
        self.df['polarity'] = [TextBlob(tweet).sentiment[0] for tweet in self.df['text']]
        self.df['subjectivity'] = [TextBlob(tweet).sentiment[1] for tweet in self.df['text']]
       
        self.df = self.df
        
        return self.df

    def send_to_sql(self, table_name):
        '''Converts internal dataframe to strings and uploads to sql database'''
        engine = create_engine(f'mysql+pymysql://{username}:{password}@{hostname}/{db_name}', encoding='utf-8')
        dfs = self.df.applymap(lambda x: str(x).encode('utf-8','ignore'))
        dfs.to_sql(table_name, engine, if_exists='append')
        
    def get_sql(self, table_name):
        engine = create_engine(f'mysql+pymysql://{username}:{password}@{hostname}/{db_name}', encoding='utf-8')
        query = f'select * from {table_name}'
        df = pd.read_sql(query, engine, index_col = 'index')
        return df

    def append_to_database(self, source, target):
        '''Adds data from source pickle to target pickle. Does processing: Remove duplicates'''
        self.df = pd.read_pickle(source)
        target_temp = pd.read_pickle(target)
        tdf = target_temp.append(self.df)
        tdf['compare'] = tdf['clean tokens'].apply(lambda x: ' '.join(x))
        tdf = tdf.drop_duplicates(['compare'])

        tdf.to_pickle(target)

        return tdf

    def news_append(self, source_dir, target):
        '''Adds data from source pickle to target pickle. Does processing: Remove duplicates'''
        files = listdir(source_dir)
        target_temp = pd.read_pickle(target)

        for source in files:
            self.df = pd.read_pickle(source_dir + source)
            target_temp = target_temp.append(self.df)

        tdf = target_temp
        tdf['tag'] = tdf['Source'].apply(self.get_tag)
        tdf = tdf.drop_duplicates(['Content'])

        tdf.to_pickle(target)

        return tdf

    def get_tag(self, inputstring):
        dictionary = {'https://www.dr.dk/nyheder/service/feeds/allenyheder#' : 'dr',
                      'http://feeds.tv2.dk/nyheder/rss' : 'tv2',
                      'https://www.bt.dk/nyheder/seneste/rss': 'bt',
                      'https://ekstrabladet.dk/rssfeed/all/' : 'eb',
                      'https://www.information.dk/feed' : 'information',
                      'https://ing.dk/rss/nyheder' : 'ing'}

        return dictionary[inputstring]

    def show_tags(self, filename):
        print(pd.read_pickle(filename).groupby('tag').size())

if __name__ == '__main__':
    '''Tests opening a pickle file and appending it to master database'''

    # Create dataprocessor
    dp = DataProcessor()

    # Append temporary file to database
    df = dp.process_twitter('twitter_temp.pickle')
#    temp_df = dp.append_to_database('twitter_temp.pickle', 'twitter_db.pickle')

    # Append news
#    temp_df = dp.news_append('rss_data/', 'news_db.pickle')
    dp.send_to_sql('twitter')
    
# %% End of file
