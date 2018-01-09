# DataProcessor
# Purpose: Maintains data from data sources and main database
# Details: Data engineering and duplicate removal
#
# %% Imports and definitions
import pandas as pd
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


if __name__ == '__main__':
    '''Tests opening a pickle file and appending it to master database'''

    # Create dataprocessor
    dp = DataProcessor()

    # Append temporary file to database
    temp_df = dp.append_to_database('twitter_temp.pickle', 'twitter_db.pickle')

# %% End of file
