# DataProcessor
# Purpose: Maintains data from data sources and main database
# Details: Data engineering and duplicate removal
#
# %% Imports and definitions
import pandas as pd
from os import listdir
pd.set_option('max_colwidth', 400)


class DataProcessor():
    '''Universal data processor for language data'''

    def __init__(self):
        pass

    def append_to_database(self, source, target):
        '''Adds data from source pickle to target pickle. Does processing: Remove duplicates'''
        source_temp = pd.read_pickle(source)
        target_temp = pd.read_pickle(target)
        tdf = target_temp.append(source_temp)
        tdf['compare'] = tdf['clean tokens'].apply(lambda x: ' '.join(x))
        tdf = tdf.drop_duplicates(['compare'])

        tdf.to_pickle(target)

        return tdf

    def news_append(self, source_dir, target):
        '''Adds data from source pickle to target pickle. Does processing: Remove duplicates'''
        files = listdir(source_dir)
        target_temp = pd.read_pickle(target)

        for source in files:
            source_temp = pd.read_pickle(source_dir + source)
            target_temp = target_temp.append(source_temp)

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
    temp_df = dp.append_to_database('twitter_temp.pickle', 'twitter_db.pickle')

    # Append news
    temp_df = dp.news_append('rss_data/', 'news_db.pickle')

# %% End of file
