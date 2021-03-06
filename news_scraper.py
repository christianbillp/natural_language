# %% - Imports
import feedparser
import pandas as pd
pd.set_option('max_colwidth', 400)
import re
import datetime

# %%

class NewsScraper():

    def __init__(self):
        with open('rss_sources.txt', 'r') as f:
            self.rss_sources = f.readlines()

    def cleanhtml(self, raw_html):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        return cleantext

    def get_headlines(self):
        rl = []
        for source in self.rss_sources:
            if source.startswith('#'):
                pass
            else:
                feed = feedparser.parse(source.strip('\n'))
                for entry in feed['entries']:
    #                print(self.cleanhtml(entry['summary']) + '\n')
                    rl.append([self.cleanhtml(entry['summary']), source.strip('\n')])
        self.db = pd.DataFrame(rl, columns=['Content', 'Source'])
#        return rl

    def present_data(self):
        with open('rss_summary.html', 'w') as f:
            f.write('<link rel="stylesheet" type="text/css" href="style.css">')
            f.write(self.db.to_html())
    
    def save_data(self):
        self.db.to_pickle("./rss_data/{}.pickle".format(datetime.datetime.today().strftime('%Y_%m_%d')))

# %%
ns = NewsScraper()
ns.get_headlines()
ns.save_data()
#ns.present_data()

# %%
#df = pd.read_pickle('./rss_data/2018_01_02.pickle')



